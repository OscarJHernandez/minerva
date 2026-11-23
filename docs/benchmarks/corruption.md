# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

898 corrupted calls over 156 reference calls (34 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 890 recovered without a model call (99.1%), 8 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 5 | 0 | 0 | 5 | 0 | 0 |
| derivative:leibniz | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 13 | 0 | 13 | 0 | 0 | 0 |
| shape:formula_alias | 101 | 0 | 101 | 0 | 0 | 0 |
| shape:json_as_string | 81 | 0 | 79 | 2 | 0 | 0 |
| shape:kind_wrapped | 9 | 0 | 9 | 0 | 0 | 0 |
| shape:list_as_string | 29 | 0 | 28 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| transport:code_fence | 134 | 0 | 134 | 0 | 0 | 0 |
| transport:dollar_wrapper | 134 | 0 | 134 | 0 | 0 | 0 |
| transport:doubled_backslash | 98 | 0 | 98 | 0 | 0 | 0 |
| transport:markdown_escape | 78 | 0 | 78 | 0 | 0 | 0 |
| transport:trailing_period | 106 | 0 | 106 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 101 | 0 | 101 | 0 | 0 | 0 |
