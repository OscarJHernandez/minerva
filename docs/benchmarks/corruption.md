# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

959 corrupted calls over 162 reference calls (35 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 948 recovered without a model call (98.9%), 11 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 2 | 2 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 6 | 0 | 0 | 6 | 0 | 0 |
| derivative:leibniz | 2 | 0 | 0 | 2 | 0 | 0 |
| derivative:spaced | 2 | 2 | 0 | 0 | 0 | 0 |
| derivative:upper | 2 | 2 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 18 | 0 | 18 | 0 | 0 | 0 |
| shape:formula_alias | 104 | 0 | 104 | 0 | 0 | 0 |
| shape:json_as_string | 87 | 0 | 85 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 35 | 0 | 34 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| syntax:braced_exponent | 2 | 2 | 0 | 0 | 0 | 0 |
| transport:code_fence | 140 | 0 | 140 | 0 | 0 | 0 |
| transport:dollar_wrapper | 140 | 0 | 140 | 0 | 0 | 0 |
| transport:doubled_backslash | 102 | 0 | 102 | 0 | 0 | 0 |
| transport:markdown_escape | 84 | 0 | 84 | 0 | 0 | 0 |
| transport:trailing_period | 112 | 0 | 112 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 107 | 0 | 107 | 0 | 0 | 0 |
