import Link from "next/link";
import { approvedExplorations } from "@minerva/ui/explorations";
import { Math } from "@minerva/ui/math";
import { Conversation } from "@/components/conversation";
import { StatusBadge } from "@/components/example-view";
import { Section } from "@/components/section";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { example, inputLatex, outputLatex } from "@/lib/capabilities";
import { modelLabel, RECORDING, session, settingsLine } from "@/lib/landing-sessions";
import {
  APP_URL,
  CONTRIBUTING_URL,
  exploreUrl,
  ISSUES_URL,
  OPENROUTER_URL,
  REPO_URL,
  SELF_HOSTING_URL,
} from "@/lib/links";

const RECORD_URL = `${REPO_URL}/blob/main/docs/benchmarks/landing-live.json`;
const primary = "rounded bg-accent px-5 py-2.5 font-medium text-accent-ink hover:opacity-90";
const secondary = "rounded border border-rule px-5 py-2.5 font-medium hover:bg-sunk";

const BENEFITS = [
  [
    "Think for yourself",
    "Pick the problem, the assumptions and the next step. Minerva suggests methods and checks your work; it does not do the thinking for you.",
  ],
  [
    "Check the mathematics",
    "A symbolic engine computes and checks results. Unverified claims stay labelled as such.",
  ],
  [
    "Keep your ideas private",
    "Host it with Docker or Kubernetes and a local model, and your conversations never leave your infrastructure. Or use the tools without AI.",
  ],
  [
    "Make it yours",
    "Apache-2.0 open source. Read it, modify it, report bugs, or contribute what is missing.",
  ],
];

/** An "Explore in Minerva" button exists only for an exploration whose prompt passed five live runs. */
function exploration(id: string) {
  return approvedExplorations("geometry-schwarzschild").find((e) => e.id === id);
}

/** One engine example: what went in, what came out, and how it was checked (worded from the engine's record). */
function Evidence({
  tag,
  title,
  lead,
  steps,
  checks,
}: {
  tag: string;
  title: string;
  lead: string;
  steps: [string, string][];
  checks: string[];
}) {
  return (
    <article className="flex min-w-0 flex-col rounded-lg border border-rule bg-raised p-5 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-medium uppercase tracking-wider text-muted">{tag}</span>
        <StatusBadge status="ok" label="Verified" />
      </div>
      <h4 className="mt-3 font-serif text-xl">{title}</h4>
      <p className="mt-1 text-sm text-muted">{lead}</p>
      <div className="mt-5 min-w-0 flex-1 rounded-md border border-rule bg-paper p-4">
        {steps.map(([label, latex], i) => (
          <div key={label}>
            {i > 0 && (
              <span className="my-1 block font-serif text-xl text-accent" aria-hidden="true">
                ↓
              </span>
            )}
            <p className="text-xs uppercase tracking-wide text-muted">{label}</p>
            <div className="mt-1 overflow-x-auto">
              <Math latex={latex} display />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-sm font-medium text-ink">{checks.join(" · ")}</p>
    </article>
  );
}

export default function Home() {
  const curvature = exploration("schwarzschild-curvature");
  const ricciExploration = exploration("schwarzschild-ricci");
  const kretschmannExploration = exploration("schwarzschild-kretschmann");
  const kepler = session("landing-kepler");
  const schwarzschild = session("landing-schwarzschild");
  const trig = example("simplify-trig-quotients");
  const laplace = example("telescoping-inverse-gaussian");
  const schottky = example("two-level-heat-capacity");
  const ode = example("ode-second-order-linear");

  return (
    <>
      <SiteHeader />
      <main id="top">
        <div className="mx-auto max-w-5xl px-5 pb-16 pt-14 sm:px-8 sm:pb-20 sm:pt-20">
          <p className="text-sm uppercase tracking-[0.18em] text-muted">
            Open source · Self-hostable · Your ideas stay yours
          </p>
          <h1 className="mt-4 max-w-3xl font-serif text-4xl leading-[1.05] tracking-tight sm:text-6xl">
            Explore mathematics. Own the process.
          </h1>
          <p className="mt-6 max-w-2xl font-serif text-xl leading-relaxed text-muted sm:text-2xl">
            Minerva is an open-source mathematical assistant you can run on your own machine.
            Explore ideas, try different approaches and verify results, while your work stays
            private to you.
          </p>
          <p className="mt-4 max-w-2xl text-base text-muted">
            AI should not take away your ability to figure things out. You choose what to
            investigate; Minerva suggests tools and checks the mathematics, and the discovery stays
            yours.
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
            Self-hosted with a local model, nothing leaves your infrastructure.{" "}
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

        <Section id="examples" title="Research with Minerva, one question at a time">
          <p className="max-w-3xl text-lg text-muted">
            You ask in your own words. Minerva chooses the tool, the engine computes and checks, and
            every formula says how far it was checked. Two real conversations, recorded from the
            deployed model and engine:
          </p>
          <Conversation session={kepler} runs={RECORDING.runs} />
          <div className="mt-4 flex flex-wrap items-center gap-4">
            <a href={APP_URL} className={secondary}>
              Try it in Minerva
            </a>
          </div>
          <Conversation session={schwarzschild} runs={RECORDING.runs} />
          <div className="mt-4 flex flex-wrap items-center gap-4">
            {curvature ? (
              <a href={exploreUrl(curvature.id)} className={primary}>
                {curvature.button}
              </a>
            ) : (
              <a href={APP_URL} className={secondary}>
                Try it in Minerva
              </a>
            )}
            {ricciExploration && (
              <a
                href={exploreUrl(ricciExploration.id)}
                className="text-sm underline hover:text-ink"
              >
                {ricciExploration.button}
              </a>
            )}
            {kretschmannExploration && (
              <a
                href={exploreUrl(kretschmannExploration.id)}
                className="text-sm underline hover:text-ink"
              >
                {kretschmannExploration.button}
              </a>
            )}
            <Link href="/docs#differential-geometry" className="text-sm underline hover:text-ink">
              Inspect the curvature calculation and assumptions
            </Link>
          </div>
          <p className="mt-6 max-w-3xl text-sm text-muted">
            Recorded {RECORDING.date} on the deployed chat model,{" "}
            <span className="text-ink">{modelLabel(RECORDING.model)}</span>, with{" "}
            {settingsLine(RECORDING.settings, RECORDING.limit, RECORDING.seed)}. Each conversation
            was run {RECORDING.runs} times; a run counts as passed only if every turn called the
            right tool, the engine verified the result and the reply made no false claim. A formula
            marked <span className="text-warn">unchecked</span> was written by the model and not
            verified; results in a tool box come from the engine.{" "}
            <a href={RECORD_URL} className="underline hover:text-ink">
              Every run, pass or fail
            </a>
            .
          </p>
          <div className="mt-12">
            <h3 className="font-serif text-2xl">Small calculations. Clear evidence.</h3>
            <p className="mt-2 max-w-3xl text-muted">
              Every result is checked twice, by routes that do not share code: a symbolic proof
              where one can be found, and numerical checks at sample points. Four results as the
              engine records them:
            </p>
            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <Evidence
                tag="01 · Verified rewrite"
                title="Simplify. Then prove the rewrite."
                lead="A shorter expression, checked before it is shown."
                steps={[
                  ["Start with", inputLatex(trig)],
                  ["Verified result", outputLatex(trig, "Result")],
                ]}
                checks={["Symbolic zero test passed", "numerical check at 20 of 20 points"]}
              />
              <Evidence
                tag="02 · A differential equation"
                title="Solve. Then put it back in."
                lead="A damped equation with initial values; the solution is substituted into the equation as written."
                steps={[
                  ["Equation", "\\begin{gathered} y'' + 3y' + 2y = 0 \\\\ y(0) = 1, \\; y'(0) = 0 \\end{gathered}"],
                  ["Verified solution", outputLatex(ode, "Solution")],
                ]}
                checks={[
                  "Symbolic zero test passed",
                  "numerical derivatives agree at 10 of 10 points",
                ]}
              />
              <Evidence
                tag="03 · A parameter integral"
                title="Laplace's integral, from a proved equation."
                lead="Minerva finds the differential equation the integral obeys, proves it exactly and solves it."
                steps={[
                  // the bounds are those of the example in docs/capabilities.yaml
                  [
                    "Integral",
                    `I(a) = \\int_0^{\\infty} ${inputLatex(laplace)}\\,dx, \\quad a > 0`,
                  ],
                  ["Proved relation", outputLatex(laplace, "Differential equation")],
                  ["Value", outputLatex(laplace, "Value")],
                ]}
                checks={[
                  "Equation proved exactly (symbolic zero)",
                  "value checked by numerical quadrature",
                ]}
              />
              <Evidence
                tag="04 · Statistical mechanics"
                title="A heat capacity, from the partition function."
                lead="A two-level system with gap ε > 0: one route, differentiate ln Z, no formula looked up."
                steps={[
                  ["Partition function", outputLatex(schottky, "Partition function")],
                  ["Internal energy", outputLatex(schottky, "Internal energy")],
                  [
                    "Heat capacity",
                    `C = ${outputLatex(schottky, "Partial derivative of internal energy")}`,
                  ],
                ]}
                checks={[
                  "U: symbolic zero, direct enumeration and finite differences agree",
                  "C: finite differences agree",
                ]}
              />
            </div>
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
                Inspect the code, extend the math engine, add a verified identity, or submit an
                issue or pull request. Minerva is licensed under Apache-2.0.
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
            Your own deployment keeps data in infrastructure you control when connected to a locally
            hosted model. External providers receive the conversations and tool results sent to
            them. The hosted demo also processes input on its operator&apos;s server.{" "}
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
