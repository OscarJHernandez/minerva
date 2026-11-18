import Link from "next/link";
import { Math } from "@minerva/ui/math";
import { CapabilityRow } from "@/components/capability-row";
import { MathLine, StatusBadge } from "@/components/example-view";
import { Section } from "@/components/section";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import {
  CAPABILITIES,
  example,
  inputLatex,
  inputSource,
  outputLatex,
  type Example,
} from "@/lib/capabilities";
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

const WHY = [
  [
    "Stay curious",
    "Mathematics is about more than finding the right answer. Explore unfamiliar ideas, follow different approaches, and discover connections you might otherwise miss.",
  ],
  [
    "A tool that helps you think",
    "Search for useful identities, investigate possible methods, simplify expressions, and check results. Minerva supports your reasoning without deciding which questions you should ask.",
  ],
  [
    "Your hardware. Your models. Your rules.",
    "Run Minerva on your own computer or server. Connect a model you host, use an external provider, or work with the mathematical tools without AI.",
  ],
  [
    "Make it yours",
    "Download it, inspect it, break it, and fix it. Report bugs, request features, contribute mathematical identities, or submit a pull request.",
  ],
];

const STEPS = [
  ["Parse", "Input is read by strict LaTeX and text parsers. Nothing you type is ever executed."],
  ["Compute", "SymPy does the algebra in an isolated process with hard time and memory limits."],
  [
    "Retrieve and check identities",
    "Search over the handbook proposes candidates. The engine matches each pattern, checks its conditions against your assumptions, and rewrites.",
  ],
  [
    "Verify",
    "Every result is checked numerically at assumption-respecting sample points and with a symbolic zero test, reported separately. A result that fails is not shown.",
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

const code = (ex: Example) => (
  <code className="block overflow-x-auto whitespace-pre text-sm">{inputSource(ex)}</code>
);
const tex = (latex: string) => <MathLine latex={latex} />;
/** What was typed (plain text) and, under it, the formula as the engine read it. */
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
  const powerGeneral = example("simplify-power-general");
  const powerPositive = example("simplify-power-positive");
  const riemann = example("tensor-riemann-contraction");
  const bianchi = example("tensor-bianchi");

  const available = CAPABILITIES.filter((c) => c.status === "available");
  const planned = CAPABILITIES.filter((c) => c.status === "planned");

  return (
    <>
      <SiteHeader />

      <main id="top">
        <div className="mx-auto max-w-5xl px-5 pb-20 pt-14 sm:px-8 sm:pb-28 sm:pt-20">
          <p className="text-sm uppercase tracking-[0.18em] text-muted">
            Open source · Self-hostable · Bring your own model
          </p>
          <h1 className="mt-4 max-w-3xl font-serif text-4xl leading-[1.05] tracking-tight sm:text-6xl">
            For the joy of figuring things out.
          </h1>
          <p className="mt-6 max-w-2xl font-serif text-xl leading-relaxed text-muted sm:text-2xl">
            Minerva is your own mathematical assistant, built for curious minds who love exploring
            ideas, discovering connections, and solving problems. Find useful identities, try
            different approaches, and check your reasoning without losing the pleasure of figuring
            things out for yourself.
          </p>
          <p className="mt-4 max-w-2xl text-base text-muted">
            Run it on your own hardware, connect a language model of your choice, or use the
            symbolic tools without AI. Open source, with mathematics you can check.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <a href={APP_URL} className={primary}>
              Try Minerva
            </a>
            <a href={REPO_URL} className={secondary}>
              Explore the source
            </a>
          </div>
          <p className="mt-4 text-sm text-muted">
            <a href={SELF_HOSTING_URL} className="underline hover:text-ink">
              Self-hosting guide
            </a>
            {" · "}
            <Link href="/docs" className="underline hover:text-ink">
              Read the docs
            </Link>
          </p>


        </div>

        <Section id="why" title="Why Minerva">
          <div className="grid gap-x-12 gap-y-10 md:grid-cols-2">
            {WHY.map(([t, d]) => (
              <div key={t}>
                <h3 className="font-serif text-2xl">{t}</h3>
                <p className="mt-2 max-w-md text-muted">{d}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="explore" title="The answer isn't always the interesting part">
          <div className="max-w-3xl space-y-4 text-lg text-muted">
            <p>
              Sometimes the interesting question isn&apos;t <em>what&apos;s the solution?</em> but{" "}
              <em>what happens if I try this instead?</em>
            </p>
            <p>
              Could a different identity simplify the problem? Is there a hidden symmetry? What
              assumptions am I making? Is there another way to approach it?
            </p>
            <p>
              Minerva is built for those questions. It doesn&apos;t replace the pleasure of
              working something out yourself. It helps you experiment, investigate, and discover
              something along the way.
            </p>
          </div>
          <figure className="mt-10 rounded-lg border border-rule bg-raised px-5 py-7 sm:px-8 sm:py-9">
            <figcaption className="text-xs uppercase tracking-wide text-muted">
              Explore a problem: the Kepler Lagrangian
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
              Both conserved quantities are verified: their time derivative is zero on the equations
              of motion, checked symbolically and numerically. Hidden symmetries, such as the
              Runge–Lenz vector, are not searched for, and the result says so.
            </p>
          </figure>
        </Section>

        <Section id="examples" title="Some examples of what it can do">
          <div className="grid gap-5 md:grid-cols-2">
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
            <Card
              title="Assumptions change the answer"
              badge="Assumption-aware"
              tone="ok"
              rows={[
                { label: "You type", node: typed(powerGeneral) },
                {
                  label: "No assumptions",
                  node: tex(String.raw`\text{unchanged: }${outputLatex(powerGeneral, "Read as")}`),
                },
                { label: "x positive, a real", node: tex(outputLatex(powerPositive, "Read as")) },
              ]}
              note="The exponents combine only where that is valid. For a general complex x the expression is left as it is, because the combined form would be wrong."
            />
            <Card
              title="Tensor symmetries"
              badge="Monoterm only"
              tone="warn"
              rows={[
                { label: "Riemann tensor contracted with a symmetric pair", node: code(riemann) },
                { label: "Canonical form", node: tex(outputLatex(riemann, "Canonical form")) },
                { label: "A cyclic sum, left alone", node: code(bianchi) },
                { label: "Canonical form", node: tex(outputLatex(bianchi, "Canonical form")) },
              ]}
              note="Symmetries of a single term are applied, by Butler–Portugal canonicalization. The cyclic sum vanishes by the Bianchi identity, a multi-term identity that is out of scope and reported as such."
            />
          </div>
          <p className="mt-8 text-sm text-muted">
            Derivatives, polynomial equations, Lagrangian mechanics and the handbook have examples
            too.{" "}
            <Link href="/docs" className="underline hover:text-ink">
              See every capability with its results and limits
            </Link>
            .
          </p>
        </Section>

        <Section id="capabilities" title="Capabilities">
          <div className="grid gap-12 md:grid-cols-2">
            <div>
              <h3 className="text-sm font-medium uppercase tracking-wide text-ok">Available now</h3>
              <div className="mt-2 border-t border-rule">
                {available.map((c) => (
                  <CapabilityRow key={c.id} cap={c} />
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-sm font-medium uppercase tracking-wide text-warn">Planned</h3>
              <div className="mt-2 border-t border-rule">
                {planned.map((c) => (
                  <CapabilityRow key={c.id} cap={c} />
                ))}
              </div>
            </div>
          </div>
          <p className="mt-10 max-w-2xl text-sm text-muted">
            Minerva is early. This list comes from the same record as the{" "}
            <Link href="/docs" className="underline hover:text-ink">
              docs
            </Link>
            , generated by running the code, and keeps everything not yet built labelled as planned.
          </p>
        </Section>

        <Section id="own" title="Your mathematics, on your terms">
          <div className="max-w-3xl space-y-4 text-muted">
            <p className="text-lg">
              Self-host Minerva with Docker or Kubernetes, connect your own Ollama model, or use
              the symbolic engine without a language model. You control your deployment and the
              database that stores your work.
            </p>
            <p>
              You can also connect OpenRouter. When you do, your conversation and tool results are
              sent to that external provider. A model you host yourself keeps model inference on
              your infrastructure.{" "}
              <a href={OPENROUTER_URL} className="underline hover:text-ink">
                Model setup and privacy
              </a>
              .
            </p>
            <p className="text-sm">
              The hosted demo processes your input on its operator&apos;s server. Try it to explore
              Minerva, but self-host if you need control of your data.
            </p>
          </div>
          <div className="mt-6">
            <a href={SELF_HOSTING_URL} className={secondary}>
              Self-host Minerva
            </a>
          </div>
        </Section>

        <Section id="open-source" title="Open source">
          <div className="max-w-3xl space-y-4 text-lg text-muted">
            <p>
              Download it. Break it. Fix it. Minerva is developed in the open under Apache-2.0.
              Read the code, change the tools, and make it your own.
            </p>
            <p>
              Found a bug or have an idea for a new feature? Open an issue. Want to build it
              yourself? Send a pull request. You can even contribute a mathematical identity as
              a plain YAML entry with a verification test, without writing application code.
            </p>
          </div>
          <div className="mt-6 flex flex-wrap gap-3">
            <a href={REPO_URL} className={primary}>
              Repository
            </a>
            <a href={ISSUES_URL} className={secondary}>
              Report a bug or suggest a feature
            </a>
            <a href={CONTRIBUTING_URL} className={secondary}>
              How to contribute
            </a>
          </div>
        </Section>

        <Section id="how" title="How Minerva checks the mathematics">
          <ol className="grid gap-8 md:grid-cols-2">
            {STEPS.map(([t, d], i) => (
              <li key={t} className="flex gap-4">
                <span className="font-serif text-3xl leading-none text-muted" aria-hidden="true">
                  {i + 1}
                </span>
                <div>
                  <h3 className="font-serif text-xl">{t}</h3>
                  <p className="mt-1 text-muted">{d}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="mt-10 max-w-3xl text-muted">
            <span className="font-medium text-ink">Optionally, a language model you host.</span>{" "}
            Connect a model served by Ollama and the chat plans, asks questions and explains,
            calling the tools above. It never computes and never decides correctness: results come
            from the engine, and formulas the model writes that no tool produced are marked
            unchecked. Chat is off until you connect a model, and how well it chooses tools depends
            on the model.
          </p>
        </Section>
      </main>

      <SiteFooter />
    </>
  );
}
