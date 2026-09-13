**018 · Warm-starting the quadratic form from a trained PSQT** · 2026-08-27
Luka's idea: initialise the diagonal of W from the trained linear model. For
binary x, `x_i^2 = x_i`, so the diagonal of W *is* a PSQT — the quadratic form
already contains every linear function. In the rank-r factorisation the
diagonal cannot be set independently of the off-diagonal, which is exactly why
the linear term is a separate parameter; that separation gives the warm start
somewhere to live. The model then begins *at* the linear optimum with the
quadratic part at zero (`lam` zeroed), verified by the warm-started model
reproducing the linear model's output to 7 digits.
**Result** (quadratic r=64, 8000 steps x 65536):

| step | cold | warm |
|---|---|---|
| 500 | 0.030008 | 0.025393 |
| 2000 | 0.025208 | 0.023318 |
| 8000 | **0.024080** | **0.022656** |

Warm r=64 beats the matched ReLU control (0.023571) that cold r=64 lost to.
**Caveat, not yet closed:** the warm arm consumed an extra 8000-step linear
pretrain, so this is not budget-matched. Both arms are flattening by step 8000
and warm is flattening *lower*, which suggests a better basin rather than only
a head start — but the control (cold at 16000 steps) decides it.

---
