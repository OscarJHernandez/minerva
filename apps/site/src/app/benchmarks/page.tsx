import type { Metadata } from "next";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { SuiteRow } from "@/components/suite-row";
import { BENCHMARKS, modelLine, pct } from "@/lib/benchmarks";
import { repoFile } from "@/lib/links";

export const metadata: Metadata = {
  title: "Benchmarks: how Minerva is measured — Minerva",
  description:
    "Problems with answers fixed in advance, run through the real engine, and the chat measured on a named model. Every number comes from a committed report; refusals and known gaps are counted, not hidden.",
};

const EVIDENCE: Record<string, string> = {
  symbolic_proof: "symbolic proofs",
  numeric_agreement: "numeric agreement (evidence, not proof)",
  structural: "structural checks",
  conditional: "conditional on a stated premise",
  refusal: "refusals",
};

function Target({ id }: { id: string }) {
  const t = BENCHMARKS.chat.targets.find((x) => x.id === id);
  if (!t) return null;
  const fmt = (v: number) => (id === "false_verified" ? String(v) : pct(v));
  return (
    <li className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-rule py-3">
      <span className="font-medium">{t.label}</span>
      <span className="text-sm tabular-nums">
        <span className="text-muted">
          target {id === "false_verified" ? "0" : `at least ${pct(t.target)}`} ·{" "}
        </span>
        measured <strong className="font-medium">{fmt(t.measured)}</strong> ·{" "}
        <span className={t.met ? "text-ok" : "text-warn"}>{t.met ? "met" : "not met"}</span>
      </span>
    </li>
  );
}

export default function Benchmarks() {
  const { suites, verification: v, chat, models } = BENCHMARKS;
  const problems = suites.reduce((n, s) => n + s.total, 0);
  const gaps = suites.reduce((n, s) => n + s.pending, 0);
  const full = chat.rows.find((r) => r.id === "all");
  return (
    <>
      <SiteHeader />
      <main className="mx-auto max-w-5xl px-5 pb-24 pt-12 sm:px-8 sm:pt-16">
        <p className="text-sm uppercase tracking-[0.18em] text-muted">Benchmarks</p>
        <h1 className="mt-4 max-w-3xl font-serif text-4xl leading-tight tracking-tight sm:text-5xl">
          How Minerva is measured
        </h1>
        <p className="mt-5 max-w-3xl font-serif text-lg leading-relaxed text-muted sm:text-xl">
          Every number on this page is read from a report committed to the repository; nothing is
          typed by hand. The answers were fixed before the code that is judged against them, from
          outside Minerva where possible. A refusal with a stated reason counts as correct, a wrong
          answer that is shown as verified counts against us, and known gaps are listed, not
          dropped.
        </p>

        <section aria-labelledby="engine-h" className="mt-14 border-t border-rule pt-12">
          <h2 id="engine-h" className="font-serif text-3xl tracking-tight sm:text-4xl">
            Does the engine get it right?
          </h2>
          <p className="mt-3 max-w-3xl text-muted">
            The engine computes and checks; no language model is involved in these results. Each
            problem ends in one of four ways, and they add up to the total.
          </p>
          <dl className="mt-6 grid max-w-3xl grid-cols-2 gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-rule bg-raised p-4">
              <dt className="text-sm text-muted">Problems in {suites.length} suites</dt>
              <dd className="mt-1 font-serif text-3xl tabular-nums" data-testid="problem-total">
                {problems}
              </dd>
            </div>
            <div className="rounded-lg border border-rule bg-raised p-4">
              <dt className="text-sm text-muted">Known gaps in them</dt>
              <dd className="mt-1 font-serif text-3xl tabular-nums">{gaps}</dd>
            </div>
            <div className="col-span-2 rounded-lg border border-rule bg-raised p-4 sm:col-span-1">
              <dt className="text-sm text-muted">Wrong verdicts on the checker itself</dt>
              <dd className="mt-1 font-serif text-3xl tabular-nums" data-testid="false-accepts">
                {v.false_accepts + v.false_rejects}
              </dd>
            </div>
          </dl>
          <div className="mt-6">
            {suites.map((s) => (
              <SuiteRow key={s.id} suite={s} />
            ))}
          </div>
        </section>

        <section aria-labelledby="checker-h" className="mt-14 border-t border-rule pt-12">
          <h2 id="checker-h" className="font-serif text-3xl tracking-tight sm:text-4xl">
            Does the checker catch wrong answers?
          </h2>
          <p className="mt-3 max-w-3xl text-muted">
            Every result is accepted only if a check that does not use the route that built it
            agrees. This corpus is {v.problems} problems with a known truth, including false
            identities that look true at random points (a log whose branch cut is hit only on the
            negative real axis, a sum that holds only for one size). It found a witness for{" "}
            {v.counterexamples_found} of them.
          </p>
          <p className="mt-3 max-w-3xl">
            <strong className="font-medium">{v.false_accepts}</strong> wrong answers accepted,{" "}
            <strong className="font-medium">{v.false_rejects}</strong> right answers rejected,{" "}
            {v.pending} pending and {v.known_open} known open weakness (listed in{" "}
            <a className="underline hover:text-ink" href={repoFile("docs/verification-defects.md")}>
              the defect record
            </a>
            ). Evidence behind the accepted reference results:{" "}
            {Object.entries(v.evidence)
              .filter(([, n]) => n > 0)
              .map(([k, n]) => `${n} ${EVIDENCE[k] ?? k.replaceAll("_", " ")}`)
              .join(", ")}
            . A pass means no disagreement was found, never a proof, unless a symbolic zero test
            says so.
          </p>
          <p className="mt-3 text-sm text-muted">
            <a className="underline hover:text-ink" href={repoFile(v.doc)}>
              How it is judged
            </a>
            {" · "}
            <code className="break-words rounded bg-sunk px-1.5 py-0.5 text-xs">{v.command}</code>
          </p>
        </section>

        <section aria-labelledby="chat-h" className="mt-14 border-t border-rule pt-12">
          <h2 id="chat-h" className="font-serif text-3xl tracking-tight sm:text-4xl">
            How well does a language model use the tools?
          </h2>
          <p className="mt-3 max-w-3xl text-muted">
            In the chat, a language model plans, asks and explains; it never computes and never
            decides what is correct. These runs measure how reliably one model chooses the right
            tool and builds a valid call. They depend on the model, so every row names it as its
            host reported it: weights, size, quantisation and digest, whether thinking was
            requested, and the reply limit. Reports of this kind carry no model host address.
          </p>

          {full && (
            <>
              <h3 className="mt-8 font-serif text-2xl tracking-tight">
                Measured against the targets
              </h3>
              <p className="mt-2 text-sm text-muted" data-testid="chat-model">
                {modelLine(full.model)}
              </p>
              <ul className="mt-4 max-w-3xl">
                {chat.targets.map((t) => (
                  <Target key={t.id} id={t.id} />
                ))}
              </ul>
              <p className="mt-3 max-w-3xl text-sm text-muted">
                {full.run.runs} runs of {full.run.cases} cases ({full.run.repeats} repeats, seed{" "}
                {full.run.seed}, {full.run.compute_limit_seconds} s limit on each tool call, code{" "}
                {full.run.code}, case set {full.run.cases_digest}, {full.date}). Median turn{" "}
                {full.median_seconds} s, slowest 5% {full.p95_seconds} s. Passes per repeat:{" "}
                {full.per_repeat.join(", ")}: the run-to-run noise is a few cases, so differences
                smaller than that mean nothing.
              </p>
            </>
          )}

          <h3 className="mt-10 font-serif text-2xl tracking-tight">All three suites</h3>
          <div className="mt-4 max-w-3xl">
            {chat.rows.map((r) => (
              <div key={r.id} className="border-b border-rule py-4" data-testid={`chat-${r.id}`}>
                <p className="font-medium">{r.title}</p>
                <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-5">
                  {[
                    ["Passed", `${r.passed}/${r.run.runs} (${pct(r.measured.pass_rate ?? 0)})`],
                    ["Tool selection", pct(r.measured.tool_selection ?? 0)],
                    ["Valid arguments", pct(r.measured.valid_arguments ?? 0)],
                    ["Completion", pct(r.measured.completion ?? 0)],
                    ["Wrongly verified", String(r.measured.false_verified)],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-muted">{k}</dt>
                      <dd className="tabular-nums">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>

          <h3 className="mt-10 font-serif text-2xl tracking-tight">Read this before quoting it</h3>
          <ul className="mt-3 max-w-3xl list-disc space-y-2 pl-5 text-muted">
            <li>
              These are the measurements before the reliability changes that followed. Most targets
              are not met on this model, and we say so.
            </li>
            <li>
              The cases were written by the people who fix what they measure, so they generalise
              less than a suite written by someone else. The held-out suite has only a few cases and
              was never used for tuning.
            </li>
            <li>
              The prompts and repairs were developed against traces of one model, which favours it
              over others.
            </li>
            <li>
              Other models were compared on the same cases under a protocol fixed beforehand, with
              and without thinking; the result is in the next section and the full account is in{" "}
              <a
                className="underline hover:text-ink"
                href={repoFile("docs/reliability/model-comparison.md")}
              >
                model-comparison.md
              </a>
              .
            </li>
          </ul>
          <p className="mt-6 text-sm text-muted">
            The full reports are in{" "}
            <a className="underline hover:text-ink" href={repoFile("docs/benchmarks")}>
              docs/benchmarks
            </a>{" "}
            and the experiment log in{" "}
            <a
              className="underline hover:text-ink"
              href={repoFile("docs/reliability/experiment-log.md")}
            >
              docs/reliability
            </a>
            .
          </p>
        </section>

        <section aria-labelledby="models-h" className="mt-14 border-t border-rule pt-12">
          <h2 id="models-h" className="font-serif text-3xl tracking-tight sm:text-4xl">
            Which model should answer?
          </h2>
          <p className="mt-3 max-w-3xl text-muted">
            Seven local models were screened with and without thinking on the probe suite, and the
            finalists were run on the full suite and the held-out suite under a protocol written
            before the runs. Every model is named by what it is, with its digest; an operator&apos;s
            alias is never the name. Accuracy decides; speed is reported apart, and was measured on
            a host shared with other work, so it is approximate.
          </p>

          <h3 className="mt-8 font-serif text-2xl tracking-tight">The full suite</h3>
          <ul className="mt-4 max-w-3xl" data-testid="model-final">
            {models.final.map((r) => (
              <li
                key={`${r.model.name}-${r.model.thinking_requested}`}
                className="border-b border-rule py-4"
                data-testid={`model-${r.model.digest}`}
              >
                <p className="font-medium">
                  {r.model.name}
                  <span className="font-normal text-muted">
                    {" "}
                    · digest {r.model.digest} · thinking {r.model.thinking_requested ? "on" : "off"}
                  </span>
                </p>
                <p className="text-sm text-muted">{r.role}</p>
                <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
                  {[
                    ["Passed", `${r.passed}/${r.run.runs} (${pct(r.measured.pass_rate ?? 0)})`],
                    ["Tool selection", pct(r.measured.tool_selection ?? 0)],
                    ["Valid arguments", pct(r.measured.valid_arguments ?? 0)],
                    ["Completion", pct(r.measured.completion ?? 0)],
                    ["Wrongly verified", String(r.measured.false_verified)],
                    ["Held-out suite", `${r.held_out.passed}/${r.held_out.runs}`],
                    ["Median turn", `${r.median_seconds} s (approximate)`],
                    ["Slowest 5%", `${r.p95_seconds} s (approximate)`],
                  ].map(([k, val]) => (
                    <div key={k}>
                      <dt className="text-muted">{k}</dt>
                      <dd className="tabular-nums">{val}</dd>
                    </div>
                  ))}
                </dl>
                {!r.eligible && (
                  <p className="mt-2 text-sm text-warn" data-testid="not-eligible">
                    Not eligible as measured: {r.not_eligible_reason}
                  </p>
                )}
              </li>
            ))}
          </ul>
          <p className="mt-3 max-w-3xl text-sm text-muted">
            {models.final[0]?.run.runs} runs each, {models.final[0]?.run.repeats} repeats, seed{" "}
            {models.final[0]?.run.seed}, reply limit {models.final[0]?.model.reply_limit} tokens,
            code {models.final[0]?.run.code}. The raw reports are in{" "}
            <a className="underline hover:text-ink" href={repoFile(models.directory)}>
              {models.directory}
            </a>
            .
          </p>

          <h3 className="mt-10 font-serif text-2xl tracking-tight">
            The screen: probe suite, thinking off and on
          </h3>
          <ul className="mt-4 max-w-3xl" data-testid="model-screen">
            {models.screen.map((r) => (
              <li
                key={`${r.model}-${r.digest}`}
                className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-rule py-3"
              >
                <span className="font-medium">
                  {r.model}
                  <span className="font-normal text-muted"> · digest {r.digest}</span>
                </span>
                <span className="text-sm tabular-nums">
                  <span className="text-muted">thinking off </span>
                  {r.off ? `${r.off.passed}/${r.off.runs}` : "not run"}
                  <span className="text-muted"> · thinking on </span>
                  {r.on ? `${r.on.passed}/${r.on.runs}` : "not run"}
                </span>
              </li>
            ))}
          </ul>

          <h3 className="mt-10 font-serif text-2xl tracking-tight">
            What it shows, and what it does not
          </h3>
          <ul className="mt-3 max-w-3xl list-disc space-y-2 pl-5 text-muted">
            <li>
              On the full suite the model the chat now uses is the most accurate configuration
              measured, by more than the run-to-run noise, and the only one that meets the
              tool-selection target. It is also about twice as slow, and no configuration meets the
              pass-rate, valid-arguments or completion targets.
            </li>
            <li>
              The screen on the small probe suite over-predicted for one finalist: it passed about
              91% of the probe runs and 62% of the full suite. A suite written by the people fixing
              what it measures is a screen, not a ranking.
            </li>
            <li>
              Thinking helped the three Qwen models, made no measurable difference for the Gemma and
              GLM builds, and hurt the Granite model. It cannot be assumed to help.
            </li>
            <li>
              The false-verified results in the table above and in the earlier runs were not a model
              failing. Two were defects in Minerva&apos;s own tools (a declared matrix read as
              commuting numbers), found because a different model wrote different prose; both are
              fixed, with tests, and one affected run is shown as not eligible until its full suite
              is repeated on the fixed code.
            </li>
            <li>
              Two models were not completed (
              <code className="rounded bg-sunk px-1 text-xs">magistral:24b</code>,{" "}
              <code className="rounded bg-sunk px-1 text-xs">ministral-3:14b</code>) and are not
              ranked. The prompts and repairs were developed on traces of one model, which favours
              it, and the weights measured here may not be byte-identical to those of a deployment
              with the same name.
            </li>
          </ul>
          <p className="mt-4 text-sm text-muted">
            The protocol, the results, the recommendation and every change made along the way are in{" "}
            <a className="underline hover:text-ink" href={repoFile(models.doc)}>
              {models.doc}
            </a>
            .
          </p>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
