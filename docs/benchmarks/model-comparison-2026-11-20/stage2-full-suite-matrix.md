# Model comparison

3 configuration(s) from 3 report(s). Case sets 091eb4bf23e4 (a row is comparable with another only on the same case set); seeds 7. Decision rule: no false verified result and fully on the GPU; then pass rate where the gap exceeds the run-to-run noise; ties go to the lower p95. Speed is reported apart: the best at any speed, and the best within the chat ceiling (median <= 15 s, p95 <= 40 s per turn on this host).

## Configurations

| Report | Model | Weights as the host reports them | Digest | Thinking | Reply budget | On GPU | Runs | Passed | Per repeat | Tool selection | Valid arguments | Completion | False verified | No tool called | Cut off | Median s | p95 s | Tokens out | Eligible | Within chat ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gemma4:26b (served as minerva) | based on gemma4:26b, 25.2B, Q4_K_M | 149a0ff8a1ca | off | 4096 | 100% | 510 | 372 (72.9%) | [126, 125, 121] | 87.3% | 88.0% | 93.7% | 0 | 20 | 1 | 6.1 | 18.9 | 108645 | yes | yes |
| 2 | glm-4.7-flash | based on glm-4.7-flash:latest, 29.9B, Q4_K_M | 4475827791a2 | off | 4096 | 100% | 510 | 316 (62.0%) | [108, 104, 104] | 86.1% | 82.5% | 83.3% | 0 | 9 | 0 | 10.2 | 51.0 | 113824 | yes | no: p95 51.0 s is over the 40 s ceiling |
| 3 | qwen3.5:35b | based on qwen35-validation-20261004:35b-a3b-q4, 34.7B, Q4_K_M | 72aac7c28be8 | on | 4096 | 100% | 510 | 389 (76.3%) | [128, 130, 131] | 95.3% | 83.5% | 92.4% | 2 | 0 | 1 | 12.4 | 43.8 | 254221 | no: 2 false verified result(s) | n/a |

## Result under the rule

- **At any speed**: gemma4:26b (served as minerva) (thinking off), digest 149a0ff8a1ca, reply budget 4096, median 6.1 s, p95 18.9 s.
- **Within the chat ceiling**: gemma4:26b (served as minerva) (thinking off), digest 149a0ff8a1ca, reply budget 4096, median 6.1 s, p95 18.9 s.
- Highest pass rate: gemma4:26b (served as minerva) (thinking off), digest 149a0ff8a1ca, 372/510 (72.9%), median 6.1 s. The result above names the fastest of the configurations that cannot be told apart from the most accurate, so a slower one that is no worse never loses to a faster one that is merely tied.

Stage-1 cut (every eligible configuration within noise of third place, at any speed): gemma4:26b (served as minerva) (thinking off), glm-4.7-flash (thinking off).

