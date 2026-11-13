# Model evaluation

> **Status of this page.** The table and failures below were produced on 2026-11-11 by the previous version
> of `scripts/eval_models.py`: 21 scored cases, one run per cell, an older claim checker. They are
> **historical and outdated**: the case set, the scoring, the claim checker and the agent loop have all
> changed since (the old `unsupported-derivative` case demanded a refusal for something `differentiate` now
> does). Do not compare them with new runs. This page is now written by hand and is never overwritten by the
> script. Current, reproducible measurements (per-category rates, tool-selection confusion matrices, latency
> distributions, failure classes, baseline-versus-new comparisons) are produced by the benchmark described in
> [observability.md](observability.md) and kept as JSON and Markdown under `docs/benchmarks/`.

## Agent benchmark: before and after the reliability changes (2026-11-13)

First reproducible measurement with the benchmark in [observability.md](observability.md): 78 cases in 10
categories, two models on a local Ollama 0.40.3, 2 repeats, seed 7, temperature 0.2, same limits. *Baseline* is the agent
as it was before the reliability changes (commit `7b19980`, frozen, with the new benchmark code on top);
*new* is the same cases after them (commit `abaeb9f`). Full reports:
[baseline](benchmarks/agent-baseline.md), [new](benchmarks/agent-new.md),
[comparison](benchmarks/agent-comparison.md) (JSON next to each). The case file has since gained five informational
geometry and distribution cases (83 in total), so its digest no longer equals the one these runs recorded
(`e89efb8efe53`); informational cases are not scored.

| | pass rate | tool selection | valid arguments | completion | unsupported shown as verified | unnecessary calls |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4:26b, baseline | 51.3% | 75.3% | 84.9% | 98.7% | 4 | 34 |
| gemma4:26b, new | 48.1% | 77.3% | 91.9% | 94.8% | 4 | 27 |
| qwen3.5:35b, baseline | 66.2% | 89.0% | 99.3% | 99.4% | 3 | 16 |
| qwen3.5:35b, new | 66.2% | 85.1% | 99.3% | 99.4% | 3 | 21 |

Targets (tool selection >= 95%, valid arguments >= 99%, completion >= 90%, no unsupported formula shown as
verified): **met** only for completion (both models) and, for `qwen3.5:35b`, valid arguments. **Not met**: tool
selection, and the count of unsupported formulas shown as verified, for both models, before and after.

What the numbers do and do not show:

- **The changes did not raise the pass rate.** `qwen3.5:35b` is unchanged (66.2%); `gemma4:26b` is 3.2 points lower.
  With two repeats and 3 to 8 unstable cases per model, differences of a few points are within the noise of a
  sampling model, so this is "no measurable improvement", not "a regression".
- **Where something moved:** `gemma4:26b` sent fewer invalid tool calls (21 of 139 before, 10 of 124 after; valid
  arguments 84.9% to 91.9%) and repeated identical calls less (15 to 10). The new loop stops a model that keeps
  calling tools that cannot help: this shows up as 8 `unproductive_loop` failures for `gemma4:26b`, which before were
  counted as other failures or as limit hits; stopping them also lowers its completion rate, by construction.
- **The unsupported-as-verified count did not fall, and it is not mainly a model fault.** All 7 hits are the
  matrix cases (`\exp(A)\exp(B) = \exp(A+B)`). The models answered correctly ("holds only if A and B commute"),
  but the claim checker treats undeclared symbols as ordinary complex numbers, so it marks that equation `checked`.
  This is a limit of the claim checker (no matrix semantics for claims), not of the model, and it is the first
  item the evidence points to; it is not fixed here.
- **Where the models fail** (new run): tool selection is the largest class (33 of 80 failures for `gemma4:26b`,
  22 of 52 for `qwen3.5:35b`), then model honesty (mostly formulas in the reply that no tool produced). In the
  `unsupported` category the models usually answer from their own belief without calling the tool (8 of 12 runs
  for `gemma4:26b`); the benchmark counts that as a tool-selection miss because "unsupported" should be the
  engine's finding. Calculus is the weakest category for `gemma4:26b` (18%): often the model passes a different
  expression than the one asked.
- **Latency** is dominated by the model (median turn 24 to 27 s on this host, tools about 0.5 s) and did not change
  beyond noise (p95 for `qwen3.5:35b` 62 s to 72 s).
- **Not measured:** other models (`qwen3.8:27b` was slower on this host and was left out), more than 2 repeats, any
  cloud provider.

## Changes after the 2026-11-13 runs (not yet measured with a model)

These follow from reading the stored replies of that run and from calling the tools directly. **No real-model run
has been made since, so nothing here claims a better pass rate.** The case set also grew from 78 to 90 cases and the
tools from 17 to 23, so the 2026-11-13 numbers are a baseline for an older system; compare a new run only with a
fresh run of the previous commit on the same case digest.

Tool interface (what the model is told when a call fails):

- A number where the schema wants text (`series` with `point: 0`) is accepted (it was refused).
- A wrong-shaped `assumptions`, `quantities` or `matrices` reports up to three errors and the right shape.
- Fixed hints for a bare `e^x` (read as a symbol, never reinterpreted), reserved `E`/`I`, and tensor index syntax.
- An `unchanged` result says that it is the answer. `unchanged`, `unsupported` and `conditions_not_satisfied` no
  longer count towards the "unproductive" stop (they are what the engine found): four different inputs that
  all come back `unchanged` end normally. Errors, invalid calls and `not_verified` still count.
- When a call leaves out assumptions declared earlier, the model is told which symbols (it was only traced).
- Prompt: a stated matrix size is enough (rule 14 made models ask for entries), the duplicate rule number is
  fixed, and rule 21 maps request types to tools.

Scoring (these change which replies pass; re-score old reports with care):

- A formula in the reply matches a tool result when both are the same function call spelled differently
  (`\sin x`, `\sin(x)`, `\sin{\left(x \right)}`): correct answers were failing `max_unchecked: 0` on spelling.
  `\sin xy` is still not `\sin(xy)`.
- Quoting the case's own false statement so that it is shown as contradicted does not count as a failure when the
  reply refutes it (matches `reply_matches`, avoids `reply_forbids`); confirming it still does.
- The runner passes the earlier formulas of a case's history to the loop as the chat route does.

To measure (same host and settings as the baseline, three repeats):

```bash
git switch --detach <previous commit> && uv run python scripts/eval_models.py --base-url "$HOST" --models gemma4:26b,qwen3.5:35b --repeat 3 --seed 7 --label before --json-out docs/benchmarks/before.json --out docs/benchmarks/before.md
git switch - && uv run python scripts/eval_models.py --base-url "$HOST" --models gemma4:26b,qwen3.5:35b --repeat 3 --seed 7 --label after --json-out docs/benchmarks/after.json --out docs/benchmarks/after.md --compare docs/benchmarks/before.json
```

## Earlier evaluation (historical)

Generated 2026-11-11 by the earlier `scripts/eval_models.py`. Each prompt runs through the real agent loop (real tools and verification) on the configured Ollama host. A case passes when the right tool is called with valid arguments, the reply says what it should (for example that conditions or assumptions are needed), and the model does not write formulas that no tool produced where that would be wrong. Results vary from run to run; treat differences of one case as noise.

| Model | Passed | Median s/case | Slowest | Formulas the model wrote |
| --- | --- | --- | --- | --- |
| qwen3.5:35b | 14/21 | 8.1 | 22.3 | 49 unchecked, 10 contradicted |
| qwen3.8:27b | 14/21 | 18.6 | 35.2 | 52 unchecked, 10 contradicted |
| gemma4:26b | 14/21 | 3.5 | 9.7 | 43 unchecked, 4 contradicted |

## Failures

### qwen3.5:35b
- `simplify-rational`: 2 unchecked formula(s) in the reply
  - reply began: 'The expression  \\frac{x^2-1}{x-1}  simplifies to  x+1 , valid for  x \\neq 1 . This result has been verified both symbolically and numerically.\n\nIf you have a specific domain or assumptions about  x  (e.g., real, positive), let me know so I '
- `tensor-contraction`: 3 equation(s) Minerva found to be false
  - reply began: 'To determine if  S_{ab}A^{ab}  is zero, I need to check the symmetry properties of the contraction. Since  S  is symmetric ( S_{ab} = S_{ba} ) and  A  is antisymmetric ( A^{ab} = -A^{ba} ), their full contraction should vanish.\n\nLet me veri'
- `ask-when-ambiguous`: reply does not say what it should
  - reply began: "Sure! Please provide the integral you'd like help with. If it involves specific variables or conditions (e.g., limits of integration, assumptions about variables), include those as well."
- `identity-compound-argument`: 1 equation(s) Minerva found to be false
  - reply began: 'The Pythagorean identity applies directly:  \\sin^2(a) + \\cos^2(a) = 1 . For your expression with  a=2t , it simplifies to  1 .\n\nOther identities (power reduction, double-angle, exponential forms) are also applicable but result in more compl'
- `solve-reducible`: 2 equation(s) Minerva found to be false
  - reply began: "The equation  x^3 + x^2 + x + 1 = 0  has three roots:\n-  x = -1 \n-  x = i \n-  x = -i \n\nThese were found by factoring the polynomial into  (x+1)(x^2+1)=0 . The result is verified by substitution and Vieta's relations."
- `lagrangian-kepler`: 3 equation(s) Minerva found to be false
  - reply began: 'The Lagrangian  \\frac12 m(\\dot{r}^2 + r^2\\dot{\\theta}^2) + \\frac{k}{r}  yields the following equations of motion:\n\n- Radial:  m \\ddot{r} = m r \\dot{\\theta}^2 - \\frac{k}{r^2} \n- Angular:  m r (r \\ddot{\\theta} + 2 \\dot{r} \\dot{\\theta}) = 0 \n\n'
- `unsupported-derivative`: 1 equation(s) Minerva found to be false; reply does not say what it should; 4 unchecked formula(s) in the reply
  - reply began: 'To differentiate  x^3 \\sin x , I will use the product rule. Let me call a tool to perform this differentiation. The tool did not recognize the derivative notation directly. Let me try a different approach by expressing the derivative using '

### qwen3.8:27b
- `simplify-rational`: 2 unchecked formula(s) in the reply
  - reply began: 'The expression simplifies to  x+1 , verified numerically (12/12 points) and symbolically. This holds for  x \\neq 1 ; at  x=1  the original is undefined (a removable singularity).'
- `tensor-contraction`: 1 equation(s) Minerva found to be false
  - reply began: 'Yes — it vanishes. Contracting a symmetric tensor with an antisymmetric one over the same pair of indices gives zero:\n\n S_{ab}A^{ab} = 0 \n\nReason: relabeling the dummy indices  a\\leftrightarrow b  gives  S_{ab}A^{ab} = S_{ba}A^{ba} = S_{ab}'
- `identity-compound-argument`: 1 equation(s) Minerva found to be false
  - reply began: 'The most direct match is the **Pythagorean identity** (`trig.pythagorean`):\n\n \\sin^2(a) + \\cos^2(a) = 1 \n\nApplied with  a = 2t , the expression simply equals **1**.\n\nThe other listed identities (power-reduction, double-angle, exponential fo'
- `solve-reducible`: 2 equation(s) Minerva found to be false
  - reply began: "The cubic factors as  (x+1)(x^2+1)=0 , giving three roots, all verified by substitution and Vieta's relations:\n\n-  x = -1 \n-  x = i \n-  x = -i \n\nAll roots are expressible in radicals (linear and quadratic factors). No assumptions on  x  wer"
- `lagrangian-kepler`: 2 equation(s) Minerva found to be false
  - reply began: 'This is the Kepler problem (attractive potential  -k/r ).\n\n**Equations of motion**\n- Radial:  m\\ddot{r} = -\\frac{k}{r^2} + m r\\dot{\\theta}^2 \n- Angular:  m r^2\\ddot{\\theta} + 2m\\dot{r}\\dot{\\theta} = 0 \n\n**Conserved quantities (verified)**\n-'
- `unsupported-derivative`: 3 equation(s) Minerva found to be false
  - reply began: "The tool misread the input as  x^2 \\sin x  (dropping the derivative) and returned it unchanged, so I can't quote a verified result from that call.\n\nThe derivative is a standard product-rule application, but since I can't verify it with a to"
- `followup-uses-context`: 1 equation(s) Minerva found to be false
  - reply began: 'The direct explanation is the **Pythagorean identity** (`trig.pythagorean`):\n\n \\sin^2 a + \\cos^2 a = 1 \n\nIt applies for any  a  (no assumptions needed), so with  a = x  the expression is exactly  1 .\n\nThe other listed identities (power redu'

### gemma4:26b
- `simplify-rational`: 1 unchecked formula(s) in the reply
  - reply began: 'The expression simplifies to  x + 1 . \n\nNote that this simplification is valid provided that  x \\neq 1 , as the original expression is undefined at that point. Would you like to explore any specific properties or constraints for this expres'
- `sqrt-no-assumptions`: 1 equation(s) Minerva found to be false; arguments not as expected: {'assumptions': {'x': ['real']}, 'latex': '\\sqrt{x^2}'}
  - reply began: 'The expression simplifies to  |x| . \n\nNote that  \\sqrt{x^2} = x  only if  x \\ge 0 . Does your problem assume  x  is non-negative?'
- `identities-blocked`: expected one of ['find_identities', 'search_handbook', 'simplify'], called none
  - reply began: 'To provide a precise answer, I need to know the domain of  x  and  y .\n\nThe identity  \\log(xy) = \\log x + \\log y  is generally valid for positive real numbers. However, if  x  and  y  are complex numbers or negative real numbers, the identi'
- `tensor-contraction`: 1 equation(s) Minerva found to be false; expected one of ['tensor_canonicalize'], called none
  - reply began: 'To determine if  S_{ab}A^{ab} = 0 , I need to know the symmetry properties of the indices. Specifically, for the antisymmetric tensor  A , are the indices  a  and  b  the same pair that is being contracted with  S ?\n\nCould you please clarif'
- `ask-when-ambiguous`: reply does not say what it should
  - reply began: 'I would be happy to help. Please provide the integral you would like to evaluate. \n\nTo ensure I use the correct tools, please specify:\n1. The expression to be integrated.\n2. The variable of integration.\n3. Any specific limits (if it is a de'
- `lagrangian-kepler`: 2 equation(s) Minerva found to be false
  - reply began: 'The equations of motion and conserved quantities for the Lagrangian  L = \\frac{1}{2} m (\\dot{r}^2 + r^2 \\dot{\\theta}^2) + \\frac{k}{r}  are:\n\n**Equations of Motion:**\n*   For  r :  \\ddot{r} = - \\frac{k}{m r^{2}} + r \\dot{\\theta}^{2} \n*   For'
- `unsupported-derivative`: reply does not say what it should; hit the step or time limit


## Reading the results

- The three models tie at 14/21, so the pass count does not separate them; tool choice, speed and honesty do.
- **qwen3.5:35b** chose the right tool in nearly every case at a median of 8 s. Chosen as the base model.
- **gemma4:26b** is fastest (3.5 s) but skips tools for tensor and identity questions (the key failure mode for this project).
- **qwen3.8:27b** is accurate but about 2.3x slower than qwen3.5:35b.
- Several remaining "found to be false" failures (Kepler equations of motion, tensor, compound-argument identity) still need review: some are real model errors, some came from the claim checker treating every equation as an identity (conditions, definitions and equations of motion). The checker no longer does that, so counts taken before the change are an upper bound on model mistakes and need re-running.
- Not evaluated here: models outside the six on the dedicated instance; the first six-model run used an older claim checker and is not comparable.

## Does automatic identity retrieval help the model? (chains of identities, 2026-11-12)

`simplify` now searches chains of verified handbook identities by itself (see `docs/handbook-benchmark.md`), and
returns the identities it used, the evidence for each step and the identities it could not apply. The question
was whether that improves the cases above. Short answer: **it did not measurably change the existing cases with
the models available for this run, and one new failure mode showed up.** This is weak evidence, not a result.

What was run: the same `scripts/eval_models.py` (plus three informational chain cases and a fix so that a
wrongly shaped argument fails a case instead of crashing the run) on the previous commit (`origin/main`) and on
this branch, against the Ollama models on the development machine: `llama3.1:8b` and `qwen3-vl:8b`. These are
**not** the models of the table above (`qwen3.5:35b`, `qwen3.8:27b`, `gemma4:26b` are not on this machine), so the
numbers are not comparable with it. One run per cell; an 8B model flips cases between runs, so a difference of
two or three cases is noise.

| Run | Before | After | Reading |
| --- | --- | --- | --- |
| `llama3.1:8b`, all scored cases | 6 / 32 | 3 / 32 | Noise: most failures are the model sending `assumptions` as a list, which both versions reject (four cases went from pass to fail, one the other way) |
| `qwen3-vl:8b`, 8 simplify/identity cases that finished on both | 5 / 8 | 5 / 8 | Same cases fail on both: `simplify-rational` (unchecked formulas in the reply), `sqrt-no-assumptions` (passes in some runs), `identity-compound-argument`; `followup-uses-context` also fails on both when it finishes (it picks `search_handbook`) |
| `qwen3-vl:8b`, new case: cites the identities used | fail (answered from the handbook search) | pass | The result lists the identities, so the model can name them |
| `qwen3-vl:8b`, new case: asked to factor | fail (no `goal` exists) | pass | `goal` is a tool argument now |
| `qwen3-vl:8b`, new case: a conditional identity | pass | fail 2 of 3 runs | See below |

The failure mode: for `\log x + \log y` with no assumptions the tool answers `conditions_not_satisfied` and names
`x` and `y` positive as the missing conditions. In two of three runs the 8B model still wrote "the simplified form
is \log(xy) ... verified numerically at 12 sample points", which the tool never said. Minerva marks that formula
as unchecked and the card shows the real status, but the sentence is wrong. The tool text now says `result=none`
and "NOT APPLIED ... this is not a result" instead of echoing the input as the result; the rerun passed once and
failed once, so it is mitigated, not fixed. A larger model is the realistic remedy; this is the same class of error as
the "formulas the model wrote" counts in the table above.

Not measured: the other cases on `qwen3-vl:8b`. Three of the six runs on that model stalled on a case (the model
server kept generating with no end; the evaluation's per-call timeout restarts with every streamed chunk, so a
case that never stops is bounded only by the step count), and the run was cut to the cases above. A stalled case
was re-run on its own and finished normally on both versions, so the stalls were not caused by the change, but
the table above is a subset. Re-run on the models of the table at the top, on `origin/main` and on this branch,
before drawing a conclusion:

```bash
uv run python scripts/eval_models.py --base-url http://HOST:11434 --models a,b --out docs/model-evaluation.md
```
