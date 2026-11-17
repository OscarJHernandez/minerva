# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

777 corrupted calls over 140 reference calls (33 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 770 recovered without a model call (99.1%), 7 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 4 | 0 | 0 | 4 | 0 | 0 |
| derivative:leibniz | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:formula_alias | 89 | 0 | 89 | 0 | 0 | 0 |
| shape:json_as_string | 72 | 0 | 70 | 2 | 0 | 0 |
| shape:list_as_string | 25 | 0 | 24 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| transport:code_fence | 118 | 0 | 118 | 0 | 0 | 0 |
| transport:dollar_wrapper | 118 | 0 | 118 | 0 | 0 | 0 |
| transport:doubled_backslash | 84 | 0 | 84 | 0 | 0 | 0 |
| transport:markdown_escape | 64 | 0 | 64 | 0 | 0 | 0 |
| transport:trailing_period | 97 | 0 | 97 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 88 | 0 | 88 | 0 | 0 | 0 |
