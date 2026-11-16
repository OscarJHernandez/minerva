# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

710 corrupted calls over 130 reference calls (33 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 703 recovered without a model call (99.0%), 7 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:latex_letter_run | 4 | 0 | 0 | 4 | 0 | 0 |
| shape:assignment_prefix | 8 | 0 | 8 | 0 | 0 | 0 |
| shape:formula_alias | 86 | 0 | 86 | 0 | 0 | 0 |
| shape:json_as_string | 67 | 0 | 65 | 2 | 0 | 0 |
| shape:list_as_string | 22 | 0 | 21 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| transport:code_fence | 108 | 0 | 108 | 0 | 0 | 0 |
| transport:dollar_wrapper | 108 | 0 | 108 | 0 | 0 | 0 |
| transport:doubled_backslash | 75 | 0 | 75 | 0 | 0 | 0 |
| transport:markdown_escape | 56 | 0 | 56 | 0 | 0 | 0 |
| transport:trailing_period | 88 | 0 | 88 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 83 | 0 | 83 | 0 | 0 | 0 |
