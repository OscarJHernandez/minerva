# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

927 corrupted calls over 159 reference calls (34 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 917 recovered without a model call (98.9%), 10 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 6 | 0 | 0 | 6 | 0 | 0 |
| derivative:leibniz | 1 | 0 | 0 | 1 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 16 | 0 | 16 | 0 | 0 | 0 |
| shape:formula_alias | 102 | 0 | 102 | 0 | 0 | 0 |
| shape:json_as_string | 84 | 0 | 82 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 32 | 0 | 31 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| syntax:braced_exponent | 1 | 1 | 0 | 0 | 0 | 0 |
| transport:code_fence | 137 | 0 | 137 | 0 | 0 | 0 |
| transport:dollar_wrapper | 137 | 0 | 137 | 0 | 0 | 0 |
| transport:doubled_backslash | 100 | 0 | 100 | 0 | 0 | 0 |
| transport:markdown_escape | 81 | 0 | 81 | 0 | 0 | 0 |
| transport:trailing_period | 109 | 0 | 109 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 104 | 0 | 104 | 0 | 0 | 0 |
