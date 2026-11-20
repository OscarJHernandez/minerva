# Model comparison

3 configuration(s) from 3 report(s). Case sets aa9c5446ddfd (a row is comparable with another only on the same case set); seeds 7. Decision rule: no false verified result and fully on the GPU; then pass rate where the gap exceeds the run-to-run noise; ties go to the lower p95. Speed is reported apart: the best at any speed, and the best within the chat ceiling (median <= 15 s, p95 <= 40 s per turn on this host).

## Configurations

| Report | Model | Weights as the host reports them | Digest | Thinking | Reply budget | On GPU | Runs | Passed | Per repeat | Tool selection | Valid arguments | Completion | False verified | No tool called | Cut off | Median s | p95 s | Tokens out | Eligible | Within chat ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gemma4:26b (served as minerva) | based on gemma4:26b, 25.2B, Q4_K_M | 149a0ff8a1ca | off | 4096 | 100% | 18 | 17 (94.4%) | [5, 6, 6] | 100.0% | 100.0% | 100.0% | 0 | 0 | 0 | 4.7 | 12.4 | 2402 | yes | yes |
| 2 | glm-4.7-flash | based on glm-4.7-flash:latest, 29.9B, Q4_K_M | 4475827791a2 | off | 4096 | 100% | 18 | 17 (94.4%) | [6, 6, 5] | 100.0% | 100.0% | 100.0% | 0 | 0 | 0 | 11.2 | 34.1 | 2489 | yes | yes |
| 3 | qwen3.5:35b | based on qwen35-validation-20261004:35b-a3b-q4, 34.7B, Q4_K_M | 72aac7c28be8 | on | 4096 | 100% | 18 | 16 (88.9%) | [5, 5, 6] | 100.0% | 100.0% | 94.4% | 0 | 0 | 0 | 16.3 | 31.8 | 5713 | yes | no: median 16.3 s is over the 15 s ceiling |

## Result under the rule

- **At any speed**: gemma4:26b (served as minerva) (thinking off), digest 149a0ff8a1ca, reply budget 4096, median 4.7 s, p95 12.4 s. Not distinguishable from it within the run-to-run noise: glm-4.7-flash (thinking off) (digest 4475827791a2, median 11.2 s), qwen3.5:35b (thinking on) (digest 72aac7c28be8, median 16.3 s).
- **Within the chat ceiling**: gemma4:26b (served as minerva) (thinking off), digest 149a0ff8a1ca, reply budget 4096, median 4.7 s, p95 12.4 s. Not distinguishable from it within the run-to-run noise: glm-4.7-flash (thinking off) (digest 4475827791a2, median 11.2 s).
- Highest pass rate: gemma4:26b (served as minerva) (thinking off), digest 149a0ff8a1ca, 17/18 (94.4%), median 4.7 s. The result above names the fastest of the configurations that cannot be told apart from the most accurate, so a slower one that is no worse never loses to a faster one that is merely tied.

Stage-1 cut (every eligible configuration within noise of third place, at any speed): gemma4:26b (served as minerva) (thinking off), glm-4.7-flash (thinking off), qwen3.5:35b (thinking on).

