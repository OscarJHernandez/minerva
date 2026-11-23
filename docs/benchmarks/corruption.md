# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

908 corrupted calls over 157 reference calls (34 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 899 recovered without a model call (99.0%), 9 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 5 | 0 | 0 | 5 | 0 | 0 |
| derivative:leibniz | 1 | 0 | 0 | 1 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 14 | 0 | 14 | 0 | 0 | 0 |
| shape:formula_alias | 102 | 0 | 102 | 0 | 0 | 0 |
| shape:json_as_string | 82 | 0 | 80 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 30 | 0 | 29 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| syntax:braced_exponent | 1 | 1 | 0 | 0 | 0 | 0 |
| transport:code_fence | 135 | 0 | 135 | 0 | 0 | 0 |
| transport:dollar_wrapper | 135 | 0 | 135 | 0 | 0 | 0 |
| transport:doubled_backslash | 98 | 0 | 98 | 0 | 0 | 0 |
| transport:markdown_escape | 79 | 0 | 79 | 0 | 0 | 0 |
| transport:trailing_period | 107 | 0 | 107 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 102 | 0 | 102 | 0 | 0 | 0 |
