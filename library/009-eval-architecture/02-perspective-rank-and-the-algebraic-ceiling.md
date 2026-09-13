## 2. Perspective, rank, and the algebraic ceiling (`runs/persp.log`)

| arm | params in V | val | vs bar |
|---|---|---|---|
| single r=512 (deployed) | 395,010 | 0.021083 | — |
| single r=1024 | 789,250 | 0.021043 | −0.19% |
| **persp r=256** | **198,402** | 0.021101 | +0.09% |
| persp r=512 | 396,034 | 0.021053 | −0.14% |
| persp r=512, no cross reader | 395,522 | 0.021063 | −0.09% |
| antisym r=512 | 395,010 | 0.021770 | **+3.26%** |

Every structural variant is inside +/-0.2%. Three conclusions:

**Rank 512 is saturated, and there is a hard ceiling at 768.** With
`act=square` and one reader the model is `x^T (V diag(R) V^T) x`, and every
symmetric 768x768 matrix has a spectral decomposition with 768 terms — so
r >= 768 cannot express anything new. Confirmed both ways: the trained
checkpoint's collapsed W has numerical rank exactly 512 with a genuinely
signed reader (283 positive / 229 negative), and 512 -> 768 buys −0.21%
(`runs/buckets.log`) against 512 -> 1024's −0.19%, the same number.

**The model is not secretly low-rank.** Eigenvalue mass says the top 64
directions hold 99.6% of ||W||_F^2, and that is **misleading** — the
big-eigenvalue directions are ones real positions rarely excite. Truncating W
to its top k directions and re-evaluating on 262,144 val positions:

| top-k | var of the quadratic eval explained | rms error |
|---|---|---|
| 64 | 64.2% | 98 cp |
| 128 | 82.4% | 71 cp |
| 256 | 98.5% | 21 cp |
| 384 | 99.94% | 5 cp |

The quadratic part has sd 146 cp and needs ~384 directions to reproduce. So
the model spreads over most of its rank; the tail carries output variance but
almost no predictive value. **Both statements are true and the second is what
makes the curve flat.**

**The perspective tie is free** — half the feature-transformer weights for
+0.09%. It buys nothing in the deployed eval (`src/qeval.rs` collapses to one
768x768 matrix, so rank is invisible at inference) but matters for NNUE, where
accumulator width *is* the cost. The cross reader `a_us . a_them` buys nothing
(−0.09%).

**`antisym` (+3.26%)** forces `f(x) = −f(flip x)`. It is the only variant
clearly outside noise, and it bounds how much frame-asymmetric structure the
model uses at all. The obvious source is tempo, which the model already has,
so it does **not** argue for storing absolute colour.
