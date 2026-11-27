# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

1022 corrupted calls over 175 reference calls (37 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 1009 recovered without a model call (98.7%), 13 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 3 | 3 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 6 | 0 | 0 | 6 | 0 | 0 |
| derivative:leibniz | 3 | 0 | 0 | 3 | 0 | 0 |
| derivative:spaced | 3 | 3 | 0 | 0 | 0 | 0 |
| derivative:upper | 3 | 3 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 23 | 0 | 23 | 0 | 0 | 0 |
| shape:formula_alias | 108 | 0 | 108 | 0 | 0 | 0 |
| shape:json_as_string | 100 | 0 | 98 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 44 | 0 | 42 | 2 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| syntax:braced_exponent | 3 | 3 | 0 | 0 | 0 | 0 |
| transport:code_fence | 145 | 0 | 145 | 0 | 0 | 0 |
| transport:dollar_wrapper | 145 | 0 | 145 | 0 | 0 | 0 |
| transport:doubled_backslash | 104 | 0 | 104 | 0 | 0 | 0 |
| transport:markdown_escape | 89 | 0 | 89 | 0 | 0 | 0 |
| transport:trailing_period | 117 | 0 | 117 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 112 | 0 | 112 | 0 | 0 | 0 |
