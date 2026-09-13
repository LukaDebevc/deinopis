**017 · Deep-linear parameterisation: real, but 82% of it is output scale** · 2026-08-27
Luka's claim: `768 -> 16 -> 1` converges far better than `768 -> 1`. It does,
and both are the *same function class* — a composition of linear maps is one
linear map — so any difference is optimisation trajectory, not capacity.
Documented as implicit acceleration by overparameterisation (Arora, Cohen &
Hazan, ICML 2018): gradient descent on the factored form behaves like the true
gradient hit by a PSD preconditioner built from the current factors, which no
regulariser on the end-to-end weight can reproduce.
**Result A** (4000 steps x 16384, CPU):

| arm | lr=0.003 | lr=0.03 |
|---|---|---|
| 768->1 | 0.057678 | 0.051098 |
| 768->16->1 | 0.042262 | 0.031807 |
| 768->64->16->1 | 0.031806 | 0.031808 |

Three arms land on 0.0318, which is therefore the converged optimum for the
class. Raising lr 10x moved `768->16->1` from 0.0423 to 0.0318 — so much of the
fixed-lr effect is **effective step size**, not depth.
**Result B, the mechanism test** (8000 steps x 65536). Hypothesis: the output is
in centipawns, so weights must travel from a 0.02 init to O(100); Adam moves a
weight ~lr per step *additively*, while a product of factors grows
*multiplicatively*. Prediction: an output scale constant should close the gap.
Init std divided by scale so every arm starts from the same output distribution.

| arm | val loss |
|---|---|
| 768->1, scale 1, lr 0.03 | 0.046679 |
| 768->1, scale 1, lr 0.3 | 0.035829 |
| 768->1, **scale 400**, lr 0.001 | 0.034203 |
| 768->64->16->1, lr 0.001 | **0.031452** |

**Scale explains 82% of the gap.** A residual survives 300x the lr and 400x the
scale, so the factorisation effect is real on top of it.
**Decision:** set the output scale to K on *every* model — the single biggest
optimisation lever found so far, and it applies to the quadratic form too
(which had been running at scale 1, i.e. crippled). Keep the deep-linear
parameterisation for PSQT and buckets: it is free, since `DeepLinear::collapse`
multiplies the factors out to the one 768-vector at inference.

---
