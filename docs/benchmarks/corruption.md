# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

818 corrupted calls over 146 reference calls (33 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 810 recovered without a model call (99.0%), 8 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 5 | 0 | 0 | 5 | 0 | 0 |
| derivative:leibniz | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 10 | 0 | 10 | 0 | 0 | 0 |
| shape:formula_alias | 92 | 0 | 92 | 0 | 0 | 0 |
| shape:json_as_string | 77 | 0 | 75 | 2 | 0 | 0 |
| shape:list_as_string | 26 | 0 | 25 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| transport:code_fence | 124 | 0 | 124 | 0 | 0 | 0 |
| transport:dollar_wrapper | 124 | 0 | 124 | 0 | 0 | 0 |
| transport:doubled_backslash | 89 | 0 | 89 | 0 | 0 | 0 |
| transport:markdown_escape | 68 | 0 | 68 | 0 | 0 | 0 |
| transport:trailing_period | 103 | 0 | 103 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 91 | 0 | 91 | 0 | 0 | 0 |
