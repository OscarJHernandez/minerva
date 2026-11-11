# Model evaluation

Generated 2026-11-11 by `scripts/eval_models.py`. Each prompt runs through the real agent loop (real tools and verification) on the configured Ollama host. A case passes when the right tool is called with valid arguments, the reply says what it should (for example that conditions or assumptions are needed), and the model does not write formulas that no tool produced where that would be wrong. Results vary from run to run; treat differences of one case as noise.

| Model | Passed | Median s/case | Slowest | Formulas the model wrote |
| --- | --- | --- | --- | --- |
| qwen3.5:35b | 14/21 | 8.1 | 22.3 | 49 unchecked, 10 contradicted |
| qwen3.8:27b | 14/21 | 18.6 | 35.2 | 52 unchecked, 10 contradicted |
| gemma4:26b | 14/21 | 3.5 | 9.7 | 43 unchecked, 4 contradicted |

## Failures

### qwen3.5:35b
- `simplify-rational`: 2 unchecked formula(s) in the reply
  - reply began: 'The expression  \\frac{x^2-1}{x-1}  simplifies to  x+1 , valid for  x \\neq 1 . This result has been verified both symbolically and numerically.\n\nIf you have a specific domain or assumptions about  x  (e.g., real, positive), let me know so I '
- `tensor-contraction`: 3 equation(s) Minerva found to be false
  - reply began: 'To determine if  S_{ab}A^{ab}  is zero, I need to check the symmetry properties of the contraction. Since  S  is symmetric ( S_{ab} = S_{ba} ) and  A  is antisymmetric ( A^{ab} = -A^{ba} ), their full contraction should vanish.\n\nLet me veri'
- `ask-when-ambiguous`: reply does not say what it should
  - reply began: "Sure! Please provide the integral you'd like help with. If it involves specific variables or conditions (e.g., limits of integration, assumptions about variables), include those as well."
- `identity-compound-argument`: 1 equation(s) Minerva found to be false
  - reply began: 'The Pythagorean identity applies directly:  \\sin^2(a) + \\cos^2(a) = 1 . For your expression with  a=2t , it simplifies to  1 .\n\nOther identities (power reduction, double-angle, exponential forms) are also applicable but result in more compl'
- `solve-reducible`: 2 equation(s) Minerva found to be false
  - reply began: "The equation  x^3 + x^2 + x + 1 = 0  has three roots:\n-  x = -1 \n-  x = i \n-  x = -i \n\nThese were found by factoring the polynomial into  (x+1)(x^2+1)=0 . The result is verified by substitution and Vieta's relations."
- `lagrangian-kepler`: 3 equation(s) Minerva found to be false
  - reply began: 'The Lagrangian  \\frac12 m(\\dot{r}^2 + r^2\\dot{\\theta}^2) + \\frac{k}{r}  yields the following equations of motion:\n\n- Radial:  m \\ddot{r} = m r \\dot{\\theta}^2 - \\frac{k}{r^2} \n- Angular:  m r (r \\ddot{\\theta} + 2 \\dot{r} \\dot{\\theta}) = 0 \n\n'
- `unsupported-derivative`: 1 equation(s) Minerva found to be false; reply does not say what it should; 4 unchecked formula(s) in the reply
  - reply began: 'To differentiate  x^3 \\sin x , I will use the product rule. Let me call a tool to perform this differentiation. The tool did not recognize the derivative notation directly. Let me try a different approach by expressing the derivative using '

### qwen3.8:27b
- `simplify-rational`: 2 unchecked formula(s) in the reply
  - reply began: 'The expression simplifies to  x+1 , verified numerically (12/12 points) and symbolically. This holds for  x \\neq 1 ; at  x=1  the original is undefined (a removable singularity).'
- `tensor-contraction`: 1 equation(s) Minerva found to be false
  - reply began: 'Yes — it vanishes. Contracting a symmetric tensor with an antisymmetric one over the same pair of indices gives zero:\n\n S_{ab}A^{ab} = 0 \n\nReason: relabeling the dummy indices  a\\leftrightarrow b  gives  S_{ab}A^{ab} = S_{ba}A^{ba} = S_{ab}'
- `identity-compound-argument`: 1 equation(s) Minerva found to be false
  - reply began: 'The most direct match is the **Pythagorean identity** (`trig.pythagorean`):\n\n \\sin^2(a) + \\cos^2(a) = 1 \n\nApplied with  a = 2t , the expression simply equals **1**.\n\nThe other listed identities (power-reduction, double-angle, exponential fo'
- `solve-reducible`: 2 equation(s) Minerva found to be false
  - reply began: "The cubic factors as  (x+1)(x^2+1)=0 , giving three roots, all verified by substitution and Vieta's relations:\n\n-  x = -1 \n-  x = i \n-  x = -i \n\nAll roots are expressible in radicals (linear and quadratic factors). No assumptions on  x  wer"
- `lagrangian-kepler`: 2 equation(s) Minerva found to be false
  - reply began: 'This is the Kepler problem (attractive potential  -k/r ).\n\n**Equations of motion**\n- Radial:  m\\ddot{r} = -\\frac{k}{r^2} + m r\\dot{\\theta}^2 \n- Angular:  m r^2\\ddot{\\theta} + 2m\\dot{r}\\dot{\\theta} = 0 \n\n**Conserved quantities (verified)**\n-'
- `unsupported-derivative`: 3 equation(s) Minerva found to be false
  - reply began: "The tool misread the input as  x^2 \\sin x  (dropping the derivative) and returned it unchanged, so I can't quote a verified result from that call.\n\nThe derivative is a standard product-rule application, but since I can't verify it with a to"
- `followup-uses-context`: 1 equation(s) Minerva found to be false
  - reply began: 'The direct explanation is the **Pythagorean identity** (`trig.pythagorean`):\n\n \\sin^2 a + \\cos^2 a = 1 \n\nIt applies for any  a  (no assumptions needed), so with  a = x  the expression is exactly  1 .\n\nThe other listed identities (power redu'

### gemma4:26b
- `simplify-rational`: 1 unchecked formula(s) in the reply
  - reply began: 'The expression simplifies to  x + 1 . \n\nNote that this simplification is valid provided that  x \\neq 1 , as the original expression is undefined at that point. Would you like to explore any specific properties or constraints for this expres'
- `sqrt-no-assumptions`: 1 equation(s) Minerva found to be false; arguments not as expected: {'assumptions': {'x': ['real']}, 'latex': '\\sqrt{x^2}'}
  - reply began: 'The expression simplifies to  |x| . \n\nNote that  \\sqrt{x^2} = x  only if  x \\ge 0 . Does your problem assume  x  is non-negative?'
- `identities-blocked`: expected one of ['find_identities', 'search_handbook', 'simplify'], called none
  - reply began: 'To provide a precise answer, I need to know the domain of  x  and  y .\n\nThe identity  \\log(xy) = \\log x + \\log y  is generally valid for positive real numbers. However, if  x  and  y  are complex numbers or negative real numbers, the identi'
- `tensor-contraction`: 1 equation(s) Minerva found to be false; expected one of ['tensor_canonicalize'], called none
  - reply began: 'To determine if  S_{ab}A^{ab} = 0 , I need to know the symmetry properties of the indices. Specifically, for the antisymmetric tensor  A , are the indices  a  and  b  the same pair that is being contracted with  S ?\n\nCould you please clarif'
- `ask-when-ambiguous`: reply does not say what it should
  - reply began: 'I would be happy to help. Please provide the integral you would like to evaluate. \n\nTo ensure I use the correct tools, please specify:\n1. The expression to be integrated.\n2. The variable of integration.\n3. Any specific limits (if it is a de'
- `lagrangian-kepler`: 2 equation(s) Minerva found to be false
  - reply began: 'The equations of motion and conserved quantities for the Lagrangian  L = \\frac{1}{2} m (\\dot{r}^2 + r^2 \\dot{\\theta}^2) + \\frac{k}{r}  are:\n\n**Equations of Motion:**\n*   For  r :  \\ddot{r} = - \\frac{k}{m r^{2}} + r \\dot{\\theta}^{2} \n*   For'
- `unsupported-derivative`: reply does not say what it should; hit the step or time limit


## Reading the results

- The three models tie at 14/21, so the pass count does not separate them; tool choice, speed and honesty do.
- **qwen3.5:35b** chose the right tool in nearly every case at a median of 8 s. Chosen as the base model.
- **gemma4:26b** is fastest (3.5 s) but skips tools for tensor and identity questions (the key failure mode for this project).
- **qwen3.8:27b** is accurate but about 2.3x slower than qwen3.5:35b.
- Several remaining "found to be false" failures (Kepler equations of motion, tensor, compound-argument identity) still need review: some are real model errors, some are limits of the claim checker (equations that are conditions or equations of motion rather than identities). The count is an upper bound on model mistakes.
- Not evaluated here: models outside the six on the dedicated instance; the first six-model run used an older claim checker and is not comparable.
