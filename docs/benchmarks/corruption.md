# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

945 corrupted calls over 161 reference calls (34 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 935 recovered without a model call (98.9%), 10 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 6 | 0 | 0 | 6 | 0 | 0 |
| derivative:leibniz | 1 | 0 | 0 | 1 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 17 | 0 | 17 | 0 | 0 | 0 |
| shape:formula_alias | 103 | 0 | 103 | 0 | 0 | 0 |
| shape:json_as_string | 86 | 0 | 84 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 34 | 0 | 33 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| syntax:braced_exponent | 1 | 1 | 0 | 0 | 0 | 0 |
| transport:code_fence | 139 | 0 | 139 | 0 | 0 | 0 |
| transport:dollar_wrapper | 139 | 0 | 139 | 0 | 0 | 0 |
| transport:doubled_backslash | 102 | 0 | 102 | 0 | 0 | 0 |
| transport:markdown_escape | 83 | 0 | 83 | 0 | 0 | 0 |
| transport:trailing_period | 111 | 0 | 111 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 106 | 0 | 106 | 0 | 0 | 0 |
