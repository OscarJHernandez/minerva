# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

737 corrupted calls over 133 reference calls (33 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 730 recovered without a model call (99.1%), 7 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 4 | 0 | 0 | 4 | 0 | 0 |
| derivative:leibniz | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:formula_alias | 89 | 0 | 89 | 0 | 0 | 0 |
| shape:json_as_string | 69 | 0 | 67 | 2 | 0 | 0 |
| shape:list_as_string | 23 | 0 | 22 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| transport:code_fence | 111 | 0 | 111 | 0 | 0 | 0 |
| transport:dollar_wrapper | 111 | 0 | 111 | 0 | 0 | 0 |
| transport:doubled_backslash | 77 | 0 | 77 | 0 | 0 | 0 |
| transport:markdown_escape | 59 | 0 | 59 | 0 | 0 | 0 |
| transport:trailing_period | 90 | 0 | 90 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 86 | 0 | 86 | 0 | 0 | 0 |
