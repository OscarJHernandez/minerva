# Homepage examples: live evaluation

Five runs of each card's exact prompt on the deployed `minerva` model (2026-11-18, seed 7, temperature 0.2, window 32768, tools limited to 10 s as in the deployed chart, at most 6 model steps per turn). A card is shown on the home page only if all five runs were correct and engine-backed with no false verification claim (`packages/ui/fixtures/homepage-examples.json`, enforced by `services/api/tests/test_homepage_examples.py`). The expected results are the textbook values of each evaluation case (`homepage-*` in `cases.yaml`), never a model's output.

| Card | Passed | Engine-backed | False verification claims | Median s | Max s | Tools a run called |
| --- | --- | --- | --- | --- | --- | --- |
| Where did the divergence go? (`coulomb-divergence`) | 5/5 | 5/5 | 0 | 34 | 37 | load_tools, vector_calculus |
| Dirac delta with multiple roots (`delta-two-roots`) | 5/5 | 5/5 | 0 | 17 | 17 | distribution |
| Verify Stokes' theorem (`stokes-curved-surface`) | 5/5 | 5/5 | 0 | 17 | 20 | load_tools, vector_calculus |
| An impulse hidden in a derivative (`heaviside-derivative`) | 5/5 | 5/5 | 0 | 21 | 22 | distribution |
| Discover hidden symmetries (`kepler-hidden-symmetries`) | 5/5 | 5/5 | 0 | 23 | 24 | find_conserved_quantities, load_tools |
| Solve an integral using symmetry (`integral-by-symmetry`) | 5/5 | 5/5 | 0 | 18 | 19 | integrate |

Failure classes in the final run: none. Several runs make a first tool call that the engine refuses (a missing operator, a coordinate system left at its default) and correct it from the error message; the tool-result column counts those. Formulas in a reply that Minerva cannot match to a result are shown as unchecked, never as verified.

## How the cards got here (what each earlier run found)

The cards were not approved on the first runs. The same prompts, run live as the work went on:

| Run | Passed | What was found, and where it was fixed |
| --- | --- | --- |
| 1 | 10/30 | Stokes: wrong `integral`, the field declared with the surface's components. Heaviside: `\theta(t)` sent, so the user was asked to confirm the notation. Coulomb: replies over the 220-word cap (a per-case cap now). Kepler: a check that forbade the word the tool itself uses for the Jacobian-rank result. Tool descriptions and prompt rules, per-case reply cap, check regexes. |
| 2 | 19/30 | The claim checker called a correct `dx = -du` contradicted after `u = pi/2 - x` (a false positive: differentials are notation, now left unverified). |
| 3-5 | iterations of cards 1, 3, 6 | A coordinate triple sent as one string, numeric bounds without quotes, a theorem sent without its expression (a closed repair when one field of its kind is declared), `r` left in Cartesian coordinates (a hint in the error), an empty model call after `load_tools` (asked again once). |
| final | 30/30 | The run recorded here, on the code that adds this file. |

Reproduce: `scripts/eval_models.py --base-url $URL --models minerva --suite core --cases <the six homepage-* ids> --repeat 5 --seed 7 --compute-timeout 10` (the host is a deployment detail and never goes in the repository).
