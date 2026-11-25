# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

994 corrupted calls over 168 reference calls (36 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 981 recovered without a model call (98.7%), 13 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 3 | 3 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 6 | 0 | 0 | 6 | 0 | 0 |
| derivative:leibniz | 3 | 0 | 0 | 3 | 0 | 0 |
| derivative:spaced | 3 | 3 | 0 | 0 | 0 | 0 |
| derivative:upper | 3 | 3 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 21 | 0 | 21 | 0 | 0 | 0 |
| shape:formula_alias | 106 | 0 | 106 | 0 | 0 | 0 |
| shape:json_as_string | 93 | 0 | 91 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 39 | 0 | 37 | 2 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| syntax:braced_exponent | 3 | 3 | 0 | 0 | 0 | 0 |
| transport:code_fence | 143 | 0 | 143 | 0 | 0 | 0 |
| transport:dollar_wrapper | 143 | 0 | 143 | 0 | 0 | 0 |
| transport:doubled_backslash | 102 | 0 | 102 | 0 | 0 | 0 |
| transport:markdown_escape | 87 | 0 | 87 | 0 | 0 | 0 |
| transport:trailing_period | 115 | 0 | 115 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 110 | 0 | 110 | 0 | 0 | 0 |
