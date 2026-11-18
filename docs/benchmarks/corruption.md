# Corruption test

Every known model mistake applied to every reference call of the benchmark, through the same pipeline a model's call takes. Each corrupted call must give the clean call's result or fail with a typed reason; anything else is a defect.

857 corrupted calls over 151 reference calls (33 tools), mode `pr`: **0 different verified results**, 0 untyped failures; 849 recovered without a model call (99.1%), 8 diagnosed.

| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |
| --- | --- | --- | --- | --- | --- | --- |
| derivative:joined | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:latex_letter_run | 5 | 0 | 0 | 5 | 0 | 0 |
| derivative:leibniz | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:spaced | 1 | 1 | 0 | 0 | 0 | 0 |
| derivative:upper | 1 | 1 | 0 | 0 | 0 | 0 |
| shape:assignment_prefix | 12 | 0 | 12 | 0 | 0 | 0 |
| shape:formula_alias | 96 | 0 | 96 | 0 | 0 | 0 |
| shape:json_as_string | 80 | 0 | 78 | 2 | 0 | 0 |
| shape:list_as_string | 28 | 0 | 27 | 1 | 0 | 0 |
| shape:number_as_string | 5 | 5 | 0 | 0 | 0 | 0 |
| transport:code_fence | 129 | 0 | 129 | 0 | 0 | 0 |
| transport:dollar_wrapper | 129 | 0 | 129 | 0 | 0 | 0 |
| transport:doubled_backslash | 94 | 0 | 94 | 0 | 0 | 0 |
| transport:markdown_escape | 73 | 0 | 73 | 0 | 0 | 0 |
| transport:trailing_period | 106 | 0 | 106 | 0 | 0 | 0 |
| transport:unicode_look_alikes | 96 | 0 | 96 | 0 | 0 | 0 |
